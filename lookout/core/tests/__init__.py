from lookout.core import server

if not server.file.exists():
    server.fetch()
