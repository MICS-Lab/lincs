<template>
  <div>
    <h1>MR-Sort model reconstruction submission</h1>
    <div v-if="submitting">Submitting...</div>
    <div v-else>
      <p>Submitted by: <input v-model="submitted_by" placeholder="Your name"/></p>
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
      submitted_by: 'Vincent',  // @todo Load from cookie, default to null
      base_api_url,
      submitting: false,
    }
  },
  methods: {
    async submit() {
      this.submitting = true
      const result = await this.$axios.$post(
        `${this.base_api_url}/computations`,
        { submitted_by: this.submitted_by },
      )
      this.$router.push({ name: 'computations-id', params: { id: result.computation_id }})
    }
  }
}
</script>
