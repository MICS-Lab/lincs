<template>
  <div>
    <h2>MR-Sort model reconstruction submission</h2>
    <div v-if="submitting">Submitting...</div>
    <div v-else>
      <!-- @todo Use Bootstrap: display a nicer form, AND VALIDATE each field -->
      <p>Submitted by: <input v-model="computation.submitted_by" placeholder="Your name"/></p>
      <p>Description: <textarea v-model="computation.description" placeholder="Free text for your convenience"></textarea></p>
      <h3>Original model</h3>
      <p><textarea v-model="computation.original_model" rows="7" cols="40"></textarea></p>
      <input @change="load_model_file" type="file"/>
      <!-- @todo Document the syntax -->
      <p><img :src="'/ppl-dev/api/mrsort-graph?model=' + computation.original_model.replaceAll('\n', ' ')" /></p>
      <h3>Learning set generation</h3>
      <p>Number of alternatives to generate: <input v-model="computation.learning_set_size"/></p>
      <p>Pseudo-random seed: <input v-model="computation.learning_set_seed"/></p>
      <!-- @todo Add command-line for generate-learning-set -->
      <h3>Model reconstruction</h3>
      <h4>Termination criteria</h4>
      <p>Target accuracy (%): <input v-model="computation.target_accuracy_percent"/></p>
      <p>Maximum duration (s): <input v-model="computation.max_duration_seconds"/></p>
      <p>Maximum number of iterations: <input v-model="computation.max_iterations"/></p>
      <h4>Algorithm</h4>
      <p>Processor: <select v-model="computation.processor">
        <option>GPU</option>
        <option>CPU</option>
      </select></p>
      <p>Pseudo-random seed: <input v-model="computation.seed"/></p>
      <!-- @todo Add weights optimization strategy -->
      <!-- @todo Add profiles improvement strategy -->
      <!-- @todo Add command-line for learn -->
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
        description: null,
        original_model: '4\n3\n0.2 0.4 0.2 0.2\n0.6\n0.3 0.4 0.2 0.5\n0.7 0.5 0.4 0.8',
        learning_set_size: 100,
        learning_set_seed: this.randomSeed(),
        target_accuracy_percent: 99,
        max_duration_seconds: null,
        max_iterations: null,
        processor: "GPU",
        seed: this.randomSeed(),
      },
    }
  },
  methods: {
    async submit() {
      this.$cookies.set("submitter_name", this.computation.submitted_by)
      this.submitting = true
      const result = await this.$axios.$post(`${this.base_api_url}/mrsort-reconstructions`, this.computation)
      this.$router.push({ name: 'computations-id', params: { id: result.computation_id }})
    },
    load_model_file(event) {
      var reader = new FileReader();
      reader.onload = (e) => {
        this.computation.original_model = e.target.result;
      };
      reader.readAsText(event.target.files[0]);
    },
    randomSeed() {
      return Math.floor(Math.random() * 65536)
    },
  }
}
</script>
