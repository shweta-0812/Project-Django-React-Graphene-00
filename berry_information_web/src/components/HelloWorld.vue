<script lang="ts">
import { useQuery } from '@vue/apollo-composable'
import gql from 'graphql-tag'

export default {
  setup () {
    const { result, loading, error } = useQuery(gql`
      query getUsers {
        userDetails {
          id
          firstName
          lastName
          email
        }
      }
    `)

    return {
      result,
      loading,
      error,
    }
  },
}


</script>

<template>
  <div v-if="loading">Loading...</div>

  <div v-else-if="error">Error: {{ error.message }}</div>

  <ul v-else-if="result && result.userDetails">
    <li v-for="user of result.userDetails" :key="user.id">
      {{ user.id }} {{ user.firstName }} {{ user.lastName }} - {{ user.email }}
    </li>
  </ul>
</template>

<style scoped>
h1 {
  font-weight: 500;
  font-size: 2.6rem;
  position: relative;
  top: -10px;
}

h3 {
  font-size: 1.2rem;
}

.greetings h1,
.greetings h3 {
  text-align: center;
}

@media (min-width: 1024px) {
  .greetings h1,
  .greetings h3 {
    text-align: left;
  }
}
</style>
