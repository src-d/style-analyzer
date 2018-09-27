from contextlib import contextmanager
from datetime import datetime
import logging
import os
import threading
from typing import Tuple, Optional

import cachetools
from pympler.asizeof import asizeof
from sqlalchemy import create_engine, Column, String, VARCHAR, DateTime, bindparam, and_
from sqlalchemy.ext import baked
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy_utils import database_exists, create_database

from lookout.core.analyzer import AnalyzerModel
from lookout.core.model_repository import ModelRepository
from lookout.core.ports import Type

Base = declarative_base()


class Model(Base):
    __tablename__ = "models"
    analyzer = Column(String(40), primary_key=True)
    repository = Column(String(40 + 100), primary_key=True)
    path = Column(VARCHAR)
    updated = Column(DateTime(timezone=True), default=datetime.utcnow)


class ContextSessionMaker:
    """
    Adds the __enter__()/__exit__() to an SQLAlchemy session and thus automatically closes it.
    """
    def __init__(self, factory):
        self.factory = factory

    @contextmanager
    def __call__(self):
        session = self.factory()
        try:
            yield session
        finally:
            session.close()


class SQLAlchemyModelRepository(ModelRepository):
    MAX_SUBDIRS = 1024
    _log = logging.getLogger("SQLAlchemyModelRepository")

    def __init__(self, db_endpoint: str, fs_root: str, max_cache_mem: int, ttl: int,
                 engine_kwargs: dict=None):
        self.fs_root = fs_root
        must_initialize = not database_exists(db_endpoint)
        if must_initialize:
            self._log.debug("%s does not exist, creating")
            create_database(db_endpoint)
            self._log.warning("created a new database at %s", db_endpoint)
        self._engine = create_engine(
            db_endpoint, **(engine_kwargs if engine_kwargs is not None else {}))
        must_initialize |= not self._engine.has_table(Model.__tablename__)
        if must_initialize:
            Model.metadata.create_all(self._engine)
        self._sessionmaker = ContextSessionMaker(sessionmaker(bind=self._engine))
        bakery = baked.bakery()
        self._get_query = bakery(lambda session: session.query(Model))
        self._get_query += lambda query: query.filter(
            and_(Model.analyzer == bindparam("analyzer"),
                 Model.repository == bindparam("repository")))
        self._cache = cachetools.TTLCache(maxsize=max_cache_mem, ttl=ttl, getsizeof=asizeof)
        self._cache_lock = threading.Lock()

    def __repr__(self) -> str:
        return "SQLAlchemyModelRepository(db_endpoint=%r, fs_root=%r, max_cache_mem=%r, " \
               "ttl=%r)" % (self._engine.url, self.fs_root, self._cache.maxsize, self._cache.ttl)

    def __str__(self) -> str:
        return "SQLAlchemyModelRepository(db=%s, fs=%s)" % (self._engine.url, self.fs_root)

    def get(self, model_id: str, model_type: Type[AnalyzerModel],
            url: str) -> Tuple[Optional[AnalyzerModel], bool]:
        cache_key = self.cache_key(model_id, model_type, url)
        with self._cache_lock:
            model = self._cache.get(cache_key)
        if model is not None:
            self._log.debug("used cache for %s with %s", model_id, url)
            return model, False
        with self._sessionmaker() as session:
            models = self._get_query(session).params(analyzer=model_id, repository=url).all()
        if len(models) == 0:
            self._log.debug("no models found for %s with %s", model_id, url)
            return None, True
        model = model_type().load(models[0].path)
        with self._cache_lock:
            self._cache[cache_key] = model
        self._log.debug("loaded %s with %s from %s", model_id, url, models[0].path)
        return model, True

    def set(self, model_id: str, url: str, model: AnalyzerModel):
        path = self.store_model(model, model_id, url)
        with self._sessionmaker() as session:
            session.merge(Model(analyzer=model_id, repository=url, path=path))
            session.commit()
        self._log.debug("set %s with %s", model_id, url)

    def init(self):
        self._log.info("initializing")
        Model.metadata.drop_all(self._engine)
        Model.metadata.create_all(self._engine)
        os.makedirs(self.fs_root, exist_ok=True)

    def shutdown(self):
        self._log.debug("shutting down")
        self._cache.clear()
        self._engine.dispose()

    @staticmethod
    def split_url(url: str):
        if url.endswith(".git"):
            url = url[:-4]
        return url[url.find("://") + 3:].split("/")

    @staticmethod
    def cache_key(model_id: str, model_type: Type[AnalyzerModel], url: str):
        return model_id + "_" + model_type.__name__ + "_" + url

    def store_model(self, model: AnalyzerModel, model_id: str, url: str) -> str:
        url_parts = self.split_url(url)
        if url_parts[0] == "github" or url_parts[0] == "bitbucket":
            url_parts = url_parts[:2] + [url_parts[2][:2]] + url_parts[2:]
        path = os.path.join(self.fs_root, *url_parts, "%s.asdf" % model_id.replace("/", "_"))
        model.save(path)
        return path
