from lm_transformer.models.lm_transformer_detail_model import LmTransformerDetail


def create_lm_transformer_detail(self, predicted_answer, question_asked, question_context):
    # Try and catch
    lm_transformer_detail = LmTransformerDetail(question_context=question_context, question_asked=question_asked,
                                                predicted_answer=predicted_answer)
    lm_transformer_detail.save()


def get_lm_transformer_detail(self):
    pass
