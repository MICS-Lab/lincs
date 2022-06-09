<template>
  <div>
    <!-- @todo Use Bootstrap -->
    <h2>Computation results</h2>
    <div v-if="loading">Loading...</div>
    <div v-else>
      <h3>Submission parameters</h3>
      <p>Submitted at: {{ computation.submitted_at }}</p>
      <p>Submitted by: {{ computation.submitted_by }}</p>
      <p>Description: {{ computation.description }}</p>
      <div v-if="computation.kind === 'mrsort-reconstruction'">
        <h3>Learning set generation</h3>
        <p>Number of alternatives to generate: {{ computation.learning_set_size }}</p>
        <p>Pseudo-random seed: {{ computation.learning_set_seed }}</p>
        <h3>Model reconstruction</h3>
        <h4>Termination criteria</h4>
        <p>Target accuracy: {{ computation.target_accuracy_percent }}%</p>
        <p>Maximum duration: {{ computation.max_duration_seconds === null ? '-' : `${computation.max_duration_seconds}s` }}</p>
        <p>Maximum number of iterations: {{ computation.max_iterations === null ? '-' : computation.max_iterations }}</p>
        <h4>Algorithm</h4>
        <p>Processor: {{ computation.processor }}</p>
        <p>Pseudo-random seed: {{ computation.seed }}</p>
        <!-- @todo Add weights optimization strategy -->
        <!-- @todo Add profiles improvement strategy -->
        <h3>Results</h3>
        <p>
          Status: {{ computation.status }}
          <span v-if="computation.status === 'failed'"> ({{ computation.failure_reason }})</span>
          <span v-else-if="!computationDone"> (This page automatically refreshes every {{ polling_interval }} seconds. You can also come back later)</span>
        </p>
        <p>Accuracy reached: {{ computation.accuracy_reached_percent === null ? '-' : `${computation.accuracy_reached_percent}%` }}</p>
        <p>Duration: {{ computation.duration_seconds === null ? '-' : `${computation.duration_seconds}s` }}</p>
        <h3>Reconstructed vs. original model</h3>
        <p>Original model: {{ computation.original_model }}</p>
        <p><img :src="'/ppl-dev/api/mrsort-graph?model=' + computation.original_model.replaceAll('\n', ' ')" /></p>
        <div v-if="computation.reconstructed_model !== null">
          <p>Reconstructed model: {{ computation.reconstructed_model }}</p>
          <p><img :src="'/ppl-dev/api/mrsort-graph?model=' + computation.reconstructed_model.replaceAll('\n', ' ')" /></p>
        </div>
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
      polling_interval: 10,
      loading: true,
      computation: null
    }
  },
  async fetch() {
    return await this.fetchComputation()
  },
  methods: {
    async fetchComputation() {
      this.computation = await this.$axios.$get(`${this.base_api_url}/computations/${this.$route.params.id}`)
      this.loading = false
      if (!this.computationDone) {
        setTimeout(() => this.fetchComputation(), this.polling_interval * 1000)
      }
    },
  },
  computed: {
    computationDone() {
      return this.computation.status !== 'queued' && this.computation.status !== 'in progress'
    },
  },
}
</script>
