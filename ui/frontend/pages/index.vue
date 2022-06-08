<template>
  <div>
    <h1>Parallel preference learning Demo</h1>
    <p>Intro paragraph, brief explanation with links to the glossary and repository (@todo write).</p>
    <h1>New computation</h1>
    <p><NuxtLink to="submit/mrsort-reconstruction">MR-Sort model reconstruction</NuxtLink> (from existing MR-Sort model)</p>
    <h1>Existing computations</h1>
    <div v-if="loading">Loading...</div>
    <div v-else>
      <div v-if="computations.length">
        <p v-for="computation in computations">
          {{ computation.submitted_by }}
          <NuxtLink :to="{'name': 'computations-id', 'params': {'id': computation.computation_id }}">Link</NuxtLink>
        </p>
      </div>
      <div v-else>No computations yet.</div>
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
      computations: []
    }
  },
  async fetch() {
    this.computations = await this.$axios.$get(`${this.base_api_url}/computations`)
    this.loading = false
  }
}
</script>
