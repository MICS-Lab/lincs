<template>
  <div>
    <h2>Computation results</h2>
    <p v-if="loading">Loading...</p>
    <p v-else-if="message">{{ message }}</p>
    <template v-else>
      <h3>Submission parameters</h3>
      <b-row>
        <b-col md>
          <p>Submitted at: {{ computation.submitted_at }}</p>
          <p>Submitted by: {{ computation.submitted_by }}</p>
          <p>Description: {{ computation.description }}</p>
        </b-col>
        <b-col md>
          <template v-if="computation.kind === 'mrsort-reconstruction'">
            <h4>Learning set generation</h4>
            <p>Number of alternatives to generate: {{ computation.learning_set_size }}</p>
            <p>Pseudo-random seed: {{ computation.learning_set_seed }}</p>
          </template>
        </b-col>
      </b-row>
      <template v-if="computation.kind === 'mrsort-reconstruction'">
        <h4>Model reconstruction</h4>
        <b-row>
          <b-col md>
            <h5>Termination criteria</h5>
            <p>Target accuracy: {{ computation.target_accuracy_percent }}%</p>
            <p>Maximum duration: {{ computation.max_duration_seconds === null ? '-' : `${computation.max_duration_seconds}s` }}</p>
            <p>Maximum number of iterations: {{ computation.max_iterations === null ? '-' : computation.max_iterations }}</p>
          </b-col>
          <b-col md>
            <h5>Algorithm</h5>
            <p>Processor: {{ computation.processor }}</p>
            <p>Pseudo-random seed: {{ computation.seed }}</p>
            <p>Weights optimization strategy: {{ computation.weights_optimization_strategy }}</p>
            <p>Profiles improvement strategy: {{ computation.profiles_improvement_strategy }}</p>
          </b-col>
        </b-row>
        <h3>Results</h3>
        <b-row>
          <b-col md>
            <p>
              Status: {{ computation.status }}
              <span v-if="computation.status === 'failed'"> ({{ computation.failure_reason }})</span>
              <span v-else-if="!computationDone"> (This page automatically refreshes every {{ polling_interval }} seconds. You can also come back later)</span>
            </p>
          </b-col>
          <b-col md>
            <p>Accuracy reached: {{ computation.accuracy_reached_percent === null ? '-' : `${computation.accuracy_reached_percent}%` }}</p>
          </b-col>
          <b-col md>
            <p>Duration: {{ computation.duration_seconds === null ? '-' : `${computation.duration_seconds}s` }}</p>
          </b-col>
        </b-row>
        <h4>Reconstructed vs. original model</h4>
        <b-row>
          <b-col md="6">
            <h5>Original model</h5>
            <pre>{{ computation.original_model }}</pre>
            <b-img fluid :src="'/ppl-dev/api/mrsort-graph?model=' + computation.original_model.replaceAll('\n', ' ')" />
          </b-col>
          <b-col md="6">
            <h5>Reconstructed model</h5>
            <template v-if="computation.reconstructed_model !== null">
              <pre>{{ computation.reconstructed_model }}</pre>
              <b-img fluid :src="'/ppl-dev/api/mrsort-graph?model=' + computation.reconstructed_model.replaceAll('\n', ' ')" />
            </template>
          </b-col>
        </b-row>
      </template>
    </template>
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
      message: null,
      computation: null,
    }
  },
  async fetch() {
    return await this.fetchComputation()
  },
  methods: {
    async fetchComputation() {
      try {
        this.computation = await this.$axios.$get(`${this.base_api_url}/computations/${this.$route.params.id}`)
      } catch {
        this.message = "Not found"
      }
      this.loading = false
      if (this.computation && !this.computationDone) {
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
