# Monitoring Log files

## Loki - Promtail
Loki is a horizontally scalable, highly available, multi-tenant log aggregation system inspired by Prometheus. It is designed to be very cost effective and easy to operate.

You can find more information about loki in this [link](https://grafana.com/oss/loki/). 

### installation
you can use [grafana's document](https://grafana.com/docs/loki/latest/installation/) to install Loki and Promtail. For Sepcific configurations with docker-compose use docker-compose.yaml file. Setup Grafana and Loki on your main server and setup Promtail on each server of your cluster. 

Log Path and Loki's URL is read from .env file. Sample.env is a template of this file. 

### Promtail configurations
In order to extract multiple labels from log lines, a pipeline stage specific to Colossalai log format is written in promtail.yaml. These labels will help with log search and Loki's performance. Add this file to your promtail config.file to use it. The sample docker-compose mounts this file to promtail's container and use it in the startup.