<template>
  <div>
    <h1>MR-Sort model reconstruction submission</h1>
    <div v-if="submitting">Submitting...</div>
    <div v-else>
      <p>Submitted by: <input v-model="computation.submitted_by" placeholder="Your name"/></p>
      <p>Description: <textarea v-model="computation.description" placeholder="Free text for your convenience"></textarea></p>
      <p><button @click="submit">Submit</button></p>
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
      submitting: false,
      computation: {
        submitted_by: typeof window === 'undefined' ? null : this.$cookies.get("submitter_name"),
        description: '',
      },
    }
  },
  methods: {
    async submit() {
      this.$cookies.set("submitter_name", this.computation.submitted_by)
      this.submitting = true
      const result = await this.$axios.$post(`${this.base_api_url}/computations`, this.computation)
      this.$router.push({ name: 'computations-id', params: { id: result.computation_id }})
    }
  }
}
</script>
