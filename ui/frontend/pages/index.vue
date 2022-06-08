<template>
  <div>
    <h1>Hello, World!</h1>
    <p>Items: {{ url }} {{ item }}</p>
  </div>
</template>

<script>
export default {
  async asyncData({ $axios  }) {
    // @todo Avoid this magic, probably by deactivating server side generation
    let url = "/ppl-dev/api/"
    if (typeof window === 'undefined') {
      url = "http://backend:8000/"
    }
    url += "items/42"

    const item = await $axios.$get(url)
    return { item, url }
  }
}
</script>
