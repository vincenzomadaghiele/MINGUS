<template>
<!-- eslint-disable --> 
    <div class="container">
        <div class="row text-center">
            <div class="col">
                <h2>Welcome to this evaluation quiz!</h2>
                <br><br>
                <h4>What is your music experience?</h4>
            </div>
        </div>
        <br>
        <div class="row text-center">
          <div class="col-lg-4"></div>
            <div class="col-lg-4">
                <div class="form-check">
                <input class="form-check-input" type="radio" name="level" value="Music lover" id="Listener" v-model="musicExperience">
                <label class="form-check-label" for="Listener">
                    <h5>Music lover</h5>
                </label>
                <p>You don't play any musical instruments 
                  or you play at an amateur level</p>
                </div>
                <div class="form-check">
                <input class="form-check-input" type="radio" name="level" value="Music student" id="musicStudent" v-model="musicExperience">
                <label class="form-check-label" for="musicStudent">
                    <h5>Music student</h5>
                </label>
                <p>You are a student in a professional-level music course</p>
                </div>
                <div class="form-check">
                <input class="form-check-input" type="radio" name="level" value="Professional musician" id="professional" v-model="musicExperience">
                <label class="form-check-label" for="professional">
                    <h5>Professional musician</h5>
                </label>
                <p>You are are a professional musician 
                  or you hold a professional level music degree
                </p>
                </div>
            </div>
            <div class="col-lg-4"></div>
        </div>
        <br><br>
        <div class="row text-center">
            <div class="col">
                <button type="button" class="btn btn-primary" @click="startQuiz">Read the quiz instructions</button>
            </div>
        </div>
        <br><br>
    </div>
</template>

<script>
import Vue from 'vue';
import VueRouter from 'vue-router';

Vue.use(VueRouter);

export default {
  data() {
    return {
      musicExperience: null,
    };
  },
  methods: {
    startQuiz() {
      if (this.musicExperience) {
        localStorage.musicExperience = this.musicExperience;
        this.$router.push('/instructions');
      } else {
        alert('Please tell us about your music experience');
      }
    },
  },
  created() {
    localStorage.musicExperience = null;
    localStorage.setItem('tunes', []);
    localStorage.setItem('rated-tunes', []);
    localStorage.setItem('comments', []);
  },
};
</script>
