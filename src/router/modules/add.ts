export default [
  {
    path: "/add",
    redirect: "/add/index",
    meta: {
      title: "添加",
      rank: 1
    },
    children: [
      {
        path: "/add/index",
        name: "Add",
        component: () => import("@/views/add/index.vue"),
        meta: {
          title: "添加"
        }
      }
    ]
  }
];
