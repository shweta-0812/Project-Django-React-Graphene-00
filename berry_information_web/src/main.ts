import './assets/main.css'
import { createApp, provide, h } from 'vue'
import App from './App.vue'
import router from './router'
import { ApolloClient, createHttpLink, InMemoryCache } from '@apollo/client/core'
import { DefaultApolloClient } from '@vue/apollo-composable'
import { createAuth0 } from '@auth0/auth0-vue';

// HTTP connection to the API
const httpLink = createHttpLink({
    // You should use an absolute URL here
    uri: 'http://localhost:8000/graphql',
})

// Cache implementation
const cache = new InMemoryCache()

// Create the apollo client
const apolloClient = new ApolloClient({
    link: httpLink,
    cache,
})

const app = createApp({
    setup () {
        provide(DefaultApolloClient, apolloClient)
    },

    render: () => h(App),
})

app.use(router)

app.use(
    createAuth0({
        domain: 'dev-ass2om-i.jp.auth0.com',
        clientId: 'YW8mCQy3UE26POlbWcJrgjhCwKF0F9W2',
        authorizationParams: {
            // redirect_uri: window.location.origin
            redirect_uri: 'https://98e3-2401-4900-1cbd-1f0a-9c84-6d4-274a-d88d.ngrok-free.app',
            audience: "http://localhost:8000/graphql",
        }
    })
)

app.mount('#app')


