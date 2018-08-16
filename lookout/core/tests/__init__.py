from lookout.core.tests import server


if not server.file.exists():
    server.fetch()
