# Security

Please report vulnerabilities privately through GitHub's **Security → Report a vulnerability** flow.
Do not include credentials, access tokens, model artifacts, or production telemetry in an issue.

The ingest API supports bearer authentication through `SPARK_CLUSTER_INGEST_TOKEN`. Set it in every
network-accessible deployment and terminate TLS at the ingress or reverse proxy.
