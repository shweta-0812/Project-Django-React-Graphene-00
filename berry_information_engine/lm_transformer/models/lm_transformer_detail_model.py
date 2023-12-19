from django.db import models


class LmTransformerDetail(models.Model):
    question_context = models.TextField()
    question_asked = models.TextField()
    predicted_answer = models.TextField()

