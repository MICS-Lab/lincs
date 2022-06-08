<template>
  <div>
    <h1>Computation results</h1>
    <div v-if="loading">Loading...</div>
    <div v-else>
      <h2>Submission parameters</h2>
      <p>Submitted at: {{ computation.submitted_at }}</p>
      <p>Submitted by: {{ computation.submitted_by }}</p>
      <p>Description: {{ computation.description }}</p>
      <div v-if="computation.kind === 'mrsort-reconstruction'">
        <h2>Learning set generation</h2>
        <p>Number of alternatives to generate: {{ computation.learning_set_size }}</p>
        <p>Pseudo-random seed: {{ computation.learning_set_seed }}</p>
        <h2>Model reconstruction</h2>
        <h3>Termination criteria</h3>
        <p>Target accuracy: {{ computation.target_accuracy_percent }}%</p>
        <p>Maximum duration: {{ computation.max_duration_seconds === null ? '-' : `${computation.max_duration_seconds}s` }}</p>
        <p>Maximum number of iterations: {{ computation.max_iterations === null ? '-' : computation.max_iterations }}</p>
        <h3>Algorithm</h3>
        <p>Processor: {{ computation.processor }}</p>
        <p>Pseudo-random seed: {{ computation.seed }}</p>
        <h2>Results</h2>
        <p>Status: {{ computation.status }}</p>
        <p>Duration: {{ computation.duration_seconds === null ? '-' : `${computation.duration_seconds}s` }}</p>
        <h2>Reconstructed vs. original model</h2>
        <p>Original model: {{ computation.original_model }}</p>
      </div>
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
