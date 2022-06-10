<template>
  <div>
    <h2>MR-Sort model reconstruction submission</h2>
    <p v-if="submitting">Submitting...</p>
    <b-form v-else @submit="submit">
      <b-form-group label="Submitted by:" label-cols-md="auto">
        <b-form-input v-model="computation.submitted_by" placeholder="Your name" type="text" required></b-form-input>
      </b-form-group>

      <b-form-group label="Description:" label-cols-md="auto">
        <b-form-textarea
          v-model="computation.description"
          placeholder="Free text for your convenience"
        ></b-form-textarea>
      </b-form-group>

      <h3>Original model</h3>
      <details>
        <summary>Model syntax</summary>
        <ul>
          <li>Line 1: a single integer, the number <i>M</i> of criteria</li>
          <li>Line 2: a single integer, the number <i>N</i> of categories</li>
          <li>Line 3: <i>M</i> decimal numbers separated by spaces, the weights of each criterion</li>
          <li>Line 4: a single decimal number, the threshold</li>
          <li>
            Lines 5 and following: <i>N - 1</i> lines, each describing a profile.
            Profiles are in increasing order, <i>i.e.</i> lowest profile first.
            Each profile consists of <i>M</i> decimal numbers separated by spaces, its values on each criterion
          </li>
        </ul>
      </details>
      <b-row>
        <b-col md>
          <b-form-textarea style="font-family: monospace;" v-model="computation.original_model" rows="11" cols="30"></b-form-textarea>
          <b-form-file v-model="original_model_file"></b-form-file>
        </b-col>
        <b-col md>
          <b-img fluid :src="'/ppl-dev/api/mrsort-graph?model=' + computation.original_model.replaceAll('\n', ' ')" />
        </b-col>
      </b-row>

      <h3>Learning set generation</h3>
      <b-form-group label="Number of alternatives to generate:" label-cols-md="auto">
        <b-form-input v-model="computation.learning_set_size" type="number" required></b-form-input>
      </b-form-group>

      <b-form-group label="Pseudo-random seed:" label-cols-md="auto">
        <b-form-input v-model="computation.learning_set_seed" type="number" required></b-form-input>
      </b-form-group>
      <!-- @todo Display command-line for generate-learning-set -->

      <h3>Model reconstruction</h3>
      <h4>Termination criteria</h4>
      <b-form-group label="Target accuracy (%):" label-cols-md="auto">
        <b-form-input v-model="computation.target_accuracy_percent" type="number" required></b-form-input>
      </b-form-group>

      <b-form-group label="Maximum duration (s):" label-cols-md="auto">
        <b-form-input v-model="computation.max_duration_seconds" type="number"></b-form-input>
      </b-form-group>

      <b-form-group label="Maximum number of iterations:" label-cols-md="auto">
        <b-form-input v-model="computation.max_iterations" type="number"></b-form-input>
      </b-form-group>

      <h4>Algorithm</h4>
      <b-form-group label="Processor:" label-cols-md="auto">
        <b-form-select v-model="computation.processor">
          <option>GPU</option>
          <option>CPU</option>
        </b-form-select>
      </b-form-group>

      <b-form-group label="Pseudo-random seed:" label-cols-md="auto">
        <b-form-input v-model="computation.seed" type="number" required></b-form-input>
      </b-form-group>

      <b-form-group label="Weights optimization strategy:" label-cols-md="auto">
        <b-form-select v-model="computation.weights_optimization_strategy" required>
          <option>glop</option>
          <option>glop-reuse</option>
        </b-form-select>
      </b-form-group>

      <b-form-group label="Profiles improvement strategy:" label-cols-md="auto">
        <b-form-select v-model="computation.profiles_improvement_strategy" required>
          <option>heuristic</option>
          <option>heuristic-midpoints</option>
        </b-form-select>
      </b-form-group>

      <!-- @todo Display command-line for learn -->
      <b-button type="submit" variant="primary">Submit</b-button>
    </b-form>
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
      original_model_from_file: '',
      original_model_file: null,
      computation: {
        submitted_by: this.$cookies && this.$cookies.get("submitted_by"),
        description: null,
        original_model: '4\n3\n0.2 0.4 0.2 0.2\n0.6\n0.3 0.4 0.2 0.5\n0.7 0.5 0.4 0.8',
        learning_set_size: 100,
        learning_set_seed: this.randomSeed(),
        target_accuracy_percent: 99,
        max_duration_seconds: null,
        max_iterations: null,
        processor: "GPU",
        seed: this.randomSeed(),
        weights_optimization_strategy: 'glop',
        profiles_improvement_strategy: 'heuristic',
      },
    }
  },
  methods: {
    async submit() {
      this.$cookies.set("submitted_by", this.computation.submitted_by, "10d", this.$router.options.base)
      this.submitting = true
      const result = await this.$axios.$post(`${this.base_api_url}/mrsort-reconstructions`, this.computation)
      this.$router.push({ name: 'computations-id', params: { id: result.computation_id }})
    },
    randomSeed() {
      return Math.floor(Math.random() * 65536)
    },
  },
  watch: {
    original_model_file() {
      if (this.original_model_file !== null) {
        var reader = new FileReader();
        reader.onload = (e) => {
          this.original_model_from_file = e.target.result
          this.computation.original_model = this.original_model_from_file
        };
        reader.readAsText(this.original_model_file);
      }
    },
    'computation.original_model'() {
      if (this.computation.original_model !== this.original_model_from_file) {
        this.original_model_file = null
      }
    },
  },
}
</script>
