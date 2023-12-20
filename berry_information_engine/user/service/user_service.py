from user.repos.user_detail_repo import create_user_detail, update_user_detail


def create_user(first_name, last_name, email):
    return create_user_detail(first_name, last_name, email)


def update_user(pk, first_name, last_name, email):
    return update_user_detail(pk, first_name, last_name, email)
