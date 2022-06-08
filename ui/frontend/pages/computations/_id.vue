<template>
  <div>
    <h1>Computation results</h1>
    <div v-if="loading">Loading...</div>
    <div v-else>
      <p>Submitted at: {{ computation.submitted_at }}</p>
      <p>Submitted by: {{ computation.submitted_by }}</p>
      <p>Description: {{ computation.description }}</p>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    // @todo Avoid this magic, probably by deactivating server side generation
    let base_api_url = "/ppl-dev/api"
    if (typeof window === 'undefined') {
      base_api_url = "http://backend:8000"
    }

    return {
      base_api_url,
      loading: true,
      computation: null
    }
  },
  async fetch() {
    this.computation = await this.$axios.$get(`${this.base_api_url}/computations/${this.$route.params.id}`)
    this.loading = false
  }
}
</script>
