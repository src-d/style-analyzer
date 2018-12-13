# GitHub integration
Short explanation how to launch style analyzer for your repository.

## Requirements
* `docker-compose` (check compatibility with compose file format [here](https://docs.docker.com/compose/compose-file/))

## How to launch
* Launch style-analyzer

`analyzer run lookout.style.format -c config_analyzer.yml`
where `config_analyzer.yml` is analyzer-specific configuration.

* Create configuration for `lookout`:
    1) Copy [docker-compose.yml.tpl](docker-compose.yml.tpl)  to `docker-compose.yml`
    2) Copy [config.yml.tpl](config.yml.tpl) to `config.yml`
    3) Update them or use environment variables

* Launch `lookout` from a directory with `docker-compose.yml` and `config.yml`
    * When `docker-compose.yml` and `config.yml` were updated

        `docker-compose up --force-recreate`

    or
    * Using environment variables
        ```
        GITHUB_USER=name GITHUB_TOKEN=tok ADDRESS=add PORT=port REPO_ORG=org  \
        REPO_NAME=repo_name docker-compose up --force-recreate
         ```