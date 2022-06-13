<template>
  <div>
    <h2>Introduction</h2>
    <p>
      <i>Parallel preference learning</i> is an attempt to provide a parallel implementation of the meta-heuristic described in <a href="https://tel.archives-ouvertes.fr/tel-01370555/document">Olivier Sobrie's thesis</a>, for learning the parameters of an MR-sort classification model.
      The <a href="https://github.com/jacquev6/parallel-preference-learning/">source code is available on GitHub</a>, as well as an introduction and a glossary explaining what it's all about.
    </p>
    <h2>New computation</h2>
    <p><NuxtLink to="submit/mrsort-reconstruction">MR-Sort model reconstruction</NuxtLink> (from existing MR-Sort model)</p>
    <h2>Existing computations</h2>
    <p v-if="loading">Loading...</p>
    <template v-else>
      <b-table v-if="computations.length" :items="computations" :fields="fields">
        <template #cell(kind)="data">
          MR-Sort model reconstruction
        </template>

        <template #cell(status)="data">
          {{ data.item.status }}<span class="d-none d-lg-inline" v-if="data.item.status === 'failed'"> ({{ data.item.failure_reason }})</span>
        </template>

        <template #cell(duration_seconds)="data">
          {{ data.item.duration_seconds === null ? '-' : `${data.item.duration_seconds}s` }}
        </template>

        <template #cell(results)="data">
          <NuxtLink :to="{'name': 'computations-id', 'params': {'id': data.item.computation_id }}">Link</NuxtLink>
        </template>
      </b-table>
      <p v-else>No computations yet.</p>
    </template>
  </div>
</template>

<script>
export default {
  data() {
    const fields = [
      {key: 'submitted_at', label: 'Submitted at'},
      {key: 'submitted_by', label: 'Submitted by'},
      'kind',
      'description',
      'status',
      {key: 'duration_seconds', label: 'Duration'},
      'results',
    ]

    return {
      loading: true,
      computations: [],
      fields
    }
  },
  async fetch() {
    this.computations = await this.$axios.$get(`computations`)
    this.loading = false
  },
}
</script>
