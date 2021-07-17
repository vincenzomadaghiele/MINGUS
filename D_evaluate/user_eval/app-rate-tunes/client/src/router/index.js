import Vue from 'vue';
import Router from 'vue-router';
import Ping from '../components/Ping.vue';
import Home from '../components/Home.vue';
import Instructions from '../components/Instructions.vue';
import Tunes from '../components/Tunes.vue';
import End from '../components/End.vue';
import Summary from '../components/Summary.vue';

Vue.use(Router);

export default new Router({
  mode: 'history',
  base: process.env.BASE_URL,
  routes: [
    {
      path: '/',
      name: 'Home',
      component: Home,
    },
    {
      path: '/instructions',
      name: 'Instructions',
      component: Instructions,
    },
    {
      path: '/tunes',
      name: 'Tunes',
      component: Tunes,
    },
    {
      path: '/ping',
      name: 'Ping',
      component: Ping,
    },
    {
      path: '/end',
      name: 'End',
      component: End,
    },
    {
      path: '/summary',
      name: 'Summary',
      component: Summary,
    },
  ],
});
