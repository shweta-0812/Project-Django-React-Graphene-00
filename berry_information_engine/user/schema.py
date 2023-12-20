import graphene
from graphene_django.types import DjangoObjectType
from user.service.user_service import create_user, update_user
from user.models.user_detail_model import UserDetail


class UserDetailType(DjangoObjectType):
    class Meta:
        model = UserDetail


class CreateUserDetail(graphene.Mutation):
    ok = graphene.Boolean()
    user_detail = graphene.Field(lambda: UserDetailType)

    class Arguments:
        first_name = graphene.String()
        last_name = graphene.String()

    def mutate(self, info, first_name, last_name, email):
        user_detail = create_user(first_name=first_name, last_name=last_name, email=email,)
        ok = True
        return CreateUserDetail(user_detail=user_detail, ok=ok)


class UpdateUserDetail(graphene.Mutation):
    user_detail = graphene.Field(lambda: UserDetailType)
    ok = graphene.Boolean()

    class Arguments:
        first_name = graphene.String()

    def mutate(self, info, id, first_name, last_name, email):
        updated_user_detail = update_user(pk=id, first_name=first_name, last_name=last_name, email=email)
        ok = True
        return UpdateUserDetail(user_detail=updated_user_detail, ok=ok)


class Query(graphene.ObjectType):
    user_details = graphene.List(UserDetailType)

    def resolve_user_details(self, info):
        return UserDetail.objects.all()


class Mutations(graphene.ObjectType):
    create_user_detail = CreateUserDetail.Field()
    update_user_detail = UpdateUserDetail.Field()