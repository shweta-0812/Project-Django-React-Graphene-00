from user.models.user_detail_model import UserDetail


def create_user_detail(first_name, last_name, email):
    user_detail = UserDetail(first_name=first_name, last_name=last_name, email=email, isDone=False)
    user_detail.save()
    return user_detail


def update_user_detail(pk, first_name, last_name, email):
    user_detail = UserDetail.objects.get(pk=pk)
    user_detail.first_name = first_name
    user_detail.last_name = last_name
    user_detail.email = email
    user_detail.save()
    return user_detail
