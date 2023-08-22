FROM postgres:13

USER postgres
COPY --chown=postgres db_init.sql /docker-entrypoint-initdb.d/

EXPOSE 5432