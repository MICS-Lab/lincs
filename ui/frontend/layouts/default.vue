<template>
  <b-container fluid="md">
    <p v-if="debug">
      Bootstrap size (debug) =
      <span class="d-sm-none">XS <i>i.e.</i> width &lt; 576px</span>
      <span class="d-none d-sm-inline d-md-none">SM <i>i.e.</i> 576px &le; width &lt; 768px</span>
      <span class="d-none d-md-inline d-lg-none">MD <i>i.e.</i> 768px &le; width &lt; 992px</span>
      <span class="d-none d-lg-inline d-xl-none">LG <i>i.e.</i> 992px &le; width &lt; 1200px</span>
      <span class="d-none d-xl-inline">XL <i>i.e.</i> 1200px &le; width</span>
    </p>
    <h1><NuxtLink :to="{name: 'index'}">Parallel preference learning Demo</NuxtLink></h1>
    <Nuxt></Nuxt>
  </b-container>
</template>

<script>
export default {
  data() {
    return {
      debug: false,
    }
  },
  mounted() {
    if (this.$cookies) {
      this.debug = (this.$cookies.get("debug_mode") || this.$route.query.debug === "1") && this.$route.query.debug !== "0"
      if (this.debug) {
        this.$cookies.set("debug_mode", true, "1h", this.$router.options.base)
      } else {
        this.$cookies.remove("debug_mode", this.$router.options.base)
      }
    }
  }
}
</script>
