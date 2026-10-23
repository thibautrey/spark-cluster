{{- define "spark-deepseek.name" -}}spark-deepseek{{- end }}
{{- define "spark-deepseek.labels" -}}
app.kubernetes.io/name: {{ include "spark-deepseek.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}
