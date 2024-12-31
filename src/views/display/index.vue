<template>
  <div>
    <el-row :gutter="20" class="mb-2">
      <el-col :span="20">
        <el-input v-model="searchQuery" placeholder="请输入震情关键词" />
      </el-col>
      <el-col :span="4">
        <el-button type="primary" @click="search">查询</el-button>
      </el-col>
    </el-row>
    <!-- 固定表头 -->
    <el-table
      :data="pagedTableData"
      class="mt-2"
      height="400"
      :row-style="{ height: '0' }"
      :cell-style="{ padding: '0' }"
    >
      <el-table-column
        v-for="column in columns"
        :key="column.prop"
        :prop="column.prop"
        :label="column.label"
        :fixed="column.prop === 'id' ? 'left' : false"
        :width="getColumnWidth(column.prop)"
      /><!-- 固定编号列 -->
      <el-table-column label="查看文件" width="100">
        <template #default="scope">
          <a
            v-if="scope.row.sourceUrl"
            :href="scope.row.sourceUrl"
            target="_blank"
            class="link-style"
            >查看文件</a
          >
        </template>
      </el-table-column>
    </el-table>
    <!-- 分页组件 -->
    <div class="pagination-container">
      <el-pagination
        background
        layout="prev, pager, next"
        :total="tableData.length"
        :page-size="15"
        @current-change="handleCurrentChange"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, computed } from "vue";
import {
  ElTable,
  ElTableColumn,
  ElInput,
  ElButton,
  ElPagination
} from "element-plus";
import "./mock.js"; // 引入 Mock 数据
import { http } from "@/utils/http";

interface EarthquakeData {
  id: number;
  location: string;
  time: string;
  source: string;
  file: string;
  category: string;
  tags: string;
  description: string;
  sourceUrl: string; // 数据源超链接
}

const tableData = ref<EarthquakeData[]>([]);
const searchQuery = ref("");

const columns = [
  { label: "编号", prop: "id" },
  { label: "参考位置", prop: "location" },
  { label: "时间", prop: "time" },
  { label: "数据源", prop: "source" },
  { label: "数据载体", prop: "file" },
  { label: "分类", prop: "category" },
  { label: "标签", prop: "tags" },
  { label: "描述", prop: "description" }
];
// 搜索请求
const search = async () => {
  try {
    const response = await http.get<{}, { data: EarthquakeData[] }>(
      "/api/earthquakes/search",
      { params: { searchKey: searchQuery.value } }
    );
    tableData.value = response.data;
  } catch (error) {
    console.error("Error searching earthquake data:", error);
  }
};
// 每15条分页
const currentPage = ref(1);
const pagedTableData = computed(() => {
  const start = (currentPage.value - 1) * 15;
  return tableData.value.slice(start, start + 15);
});
// 当前页码改变时触发
const handleCurrentChange = (val: number) => {
  currentPage.value = val;
};
// 定义列宽
const getColumnWidth = (prop: string) => {
  switch (prop) {
    case "description":
      return "300px"; // 例如，为 'id' 列设置固定宽度
    case "id":
      return "150px"; // 根据内容调整
    case "location":
      return "170px"; // 根据内容调整
    default:
      return "120px"; // 默认宽度
  }
};
// 加载时对空字符串进行搜索，返回所有数据
onMounted(async () => {
  await search();
});
</script>

<style scoped>
.mb-2 {
  margin-bottom: 10px;
}

.mt-2 {
  margin-top: 10px;
}

/* 分页标居中 */
.pagination-container {
  display: flex;
  justify-content: center;
  margin-top: 10px;
}

.link-style {
  color: rgb(102 134 198);
  cursor: pointer;
}

.link-style:hover {
  color: darkblue;
  text-decoration: none;
}
</style>
