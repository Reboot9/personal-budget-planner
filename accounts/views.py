from django.urls import reverse_lazy
from django.views.generic import CreateView, FormView, RedirectView
from django.contrib.auth import login, logout
from django.contrib.auth.views import LoginView
from .forms import CustomUserCreationForm, CustomAuthenticationForm

class RegisterView(CreateView):
    form_class = CustomUserCreationForm
    template_name = "registration/register.html"
    success_url = reverse_lazy("login")

class CustomLoginView(LoginView):
    form_class = CustomAuthenticationForm
    template_name = "registration/login.html"

    def form_valid(self, form):
        login(self.request, form.get_user())
        return super().form_valid(form)

    def get_success_url(self):
        return reverse_lazy("home")

# class LogoutView(RedirectView):
#     pattern_name = "login"
#
#     def get(self, request, *args, **kwargs):
#         logout(request)
#         return super().get(request, *args, **kwargs)
