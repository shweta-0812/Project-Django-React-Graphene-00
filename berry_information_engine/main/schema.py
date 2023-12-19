import graphene
import user.schema
import lm_transformer.schema


class Query(user.schema.Query, graphene.ObjectType):
    # This class will inherit from multiple Queries
    # as we begin to add more apps to our project
    pass


class Mutation(
    lm_transformer.schema.Mutations,
    user.schema.Mutations,
    graphene.ObjectType):
    # This class will inherit from multiple Mutations
    # as we begin to add more apps to our project
    pass


schema = graphene.Schema(query=Query, mutation=Mutation)
