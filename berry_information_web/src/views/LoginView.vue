<template>
  <main>
    <div class="container">
      <ul>
        <li v-if="!isAuthenticated && !isLoading">
          <button @click.prevent="login">Log In</button>
        </li>

        <li v-if="isAuthenticated">
          <button @click.prevent="logout">Log out</button>
        </li>

        <li v-if="isAuthenticated">
          <img
              :src="authenticatedUser?.picture"
              alt="User's profile picture"
              class="rounded-circle img-fluid profile-picture"
          />
          <pre v-if="isAuthenticated">
              <code>{{ authenticatedUser }}</code>
            </pre>
        </li>

      </ul>
    </div>
  </main>
</template>

<script lang="ts">
import { useAuth0 } from '@auth0/auth0-vue';

export default {
  name: "LoginView",
  setup() {
    const auth0 = useAuth0();

    return {
      isAuthenticated: auth0.isAuthenticated,
      isLoading: auth0.isLoading,
      authenticatedUser: auth0.user,
      getAccessTokenSilently: auth0.getAccessTokenSilently(),
      login() {
        auth0.loginWithRedirect();
      },
      logout() {
        auth0.logout({
          logoutParams: {
            // returnTo: window.location.origin
            returnTo: 'https://98e3-2401-4900-1cbd-1f0a-9c84-6d4-274a-d88d.ngrok-free.app/home'
          }
        });
      },
      doSomethingWithToken: async () => {
        const token = await getAccessTokenSilently();
      }
    }
  }
};
</script>

