import graphene
from lm_transformer.service.lm_transformer_service import get_predicted_answer_for_question_and_context


class AskLmTransformer(graphene.Mutation):
    ok = graphene.Boolean()
    question_context = graphene.String()
    question_asked = graphene.String()
    predicted_answer = graphene.String()

    class Arguments:
        question_context = graphene.String()
        question_asked = graphene.String()

    def mutate(self, info, question_context, question_asked):
        predicted_answer, question_context, question_asked = get_predicted_answer_for_question_and_context(question_context=question_context, question_asked=question_asked)
        if not predicted_answer:
            return AskLmTransformer(predicted_answer=None, question_context=question_context, question_asked=question_asked , ok=False)
        return AskLmTransformer(predicted_answer=predicted_answer, question_context=question_context, question_asked=question_asked , ok=True)


class Mutations(graphene.ObjectType):
    ask_lm_transformer = AskLmTransformer.Field()
