from datetime import datetime

import modelforge
from sqlalchemy import create_engine, Column, String, VARCHAR, DateTime, bindparam
from sqlalchemy.ext import baked
from sqlalchemy.ext.declarative import declarative_base

from lookout.core.model_repository import ModelRepository


Base = declarative_base()


class Models(Base):
    __tablename__ = "models"
    analyzer = Column(String(40), primary_key=True)
    repository = Column(String(40 + 100), primary_key=True)
    path = Column(VARCHAR)
    updated = Column(DateTime(timezone=True), default=datetime.utcnow)


class SQLAlchemyModelRepository(ModelRepository):
    def __init__(self, db_endpoint: str):
        self._engine = create_engine(db_endpoint)
        bakery = baked.bakery()
        self._get_query = bakery(lambda session: session.query(Models))
        self._get_query += lambda query: query.filter(Models.analyzer == bindparam("analyzer"))

    def get(self, model_id: str, url) -> modelforge.Model:
        raise NotImplementedError

    def set(self, model_id: str, url, model: modelforge.Model):
        raise NotImplementedError

    def init(self):
        Models.metadata.create_all(self._engine)
