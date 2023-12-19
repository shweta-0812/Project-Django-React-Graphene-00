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
        domain: 'you_auth0_domain',
        clientId: 'your_auth0_clientId',
        authorizationParams: {
            redirect_uri: window.location.origin,
            // redirect_uri: 'ngrok url in case you are running on localhost',
            audience: "http://localhost:8000/graphql",
        }
    })
)

app.mount('#app')


