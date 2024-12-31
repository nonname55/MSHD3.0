<template>
  <el-card class="box-card">
    <template #header>
      <div class="card-header">
        <span>添加项目</span>
      </div>
    </template>
    <div>
      <el-form :model="form" label-width="90px">
        <el-form-item label="街道">
          <region-selects
            :town="true"
            v-model="form.location"
            @change="onChangeRegion"
          />
        </el-form-item>
        <el-form-item label="社区">
          <el-select
            v-model="form.village"
            placeholder="选择社区"
            @visible-change="$forceUpdate()"
          >
            <el-option
              v-for="item in villageOptions"
              :key="item.value"
              :label="item.title"
              :value="item.value"
            />
          </el-select>
        </el-form-item>
        <el-form-item label="灾情大类">
          <el-select
            v-model="form.disaster.major"
            placeholder="选择灾情大类"
            @change="onChangeDisasterMajor($event)"
          >
            <el-option label="震情" value="1" />
            <el-option label="人员伤亡及失踪" value="2" />
            <el-option label="房屋破坏" value="3" />
            <el-option label="生命线工程灾情" value="4" />
            <el-option label="次生灾害" value="5" />
          </el-select>
        </el-form-item>
        <el-form-item label="灾情子类">
          <el-select v-model="form.disaster.minor" placeholder="选择灾情子类">
            <el-option
              v-for="item in minorOptions[form.disaster.major]"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
        </el-form-item>
        <el-form-item label="灾情指标">
          <el-select v-model="form.disaster.index" placeholder="选择灾情指标">
            <el-option
              v-for="item in indexOptions[form.disaster.major]"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
        </el-form-item>
        <el-form-item label="载体">
          <el-select v-model="form.carrierCode" placeholder="选择载体">
            <el-option label="文字" value="0" />
            <el-option label="图像" value="1" />
            <el-option label="音频" value="2" />
            <el-option label="视频" value="3" />
            <el-option label="其他" value="4" />
          </el-select>
        </el-form-item>
        <el-form-item label="文字信息" v-if="form.carrierCode == '0'">
          <el-input v-model="form.desc" type="textarea" />
        </el-form-item>
        <el-form-item
          label="文件信息"
          v-if="
            form.carrierCode == '1' ||
            form.carrierCode == '2' ||
            form.carrierCode == '3' ||
            form.carrierCode == '4'
          "
        >
          <el-upload
            ref="upload"
            action="/api/upload"
            :limit="1"
            :on-exceed="handleExceed"
            :auto-upload="false"
            :on-change="onChangeFile"
          >
            <template #trigger>
              <el-button type="primary">选择文件</el-button>
            </template>
            <el-button class="ml-3" type="success" @click="submitUpload">
              确认上传
            </el-button>
            <template #tip>
              <div class="el-upload__tip text-red">
                仅能上传1个文件，多文件请新建新的灾情信息
              </div>
            </template>
          </el-upload>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="onSubmit">提交</el-button>
        </el-form-item>
      </el-form>
    </div>
    <!--<pre>{{ JSON.stringify(form, null, 2) }}</pre>-->
  </el-card>
</template>

<script setup lang="ts">
import { reactive, ref } from "vue";
import { genFileId, ElMessage } from "element-plus";
import type { UploadInstance, UploadProps, UploadRawFile } from "element-plus";
import { RegionSelects } from "v-region";
import { http } from "@/utils/http";
defineOptions({
  name: "Add"
});

const form = reactive({
  location: {
    province: "",
    city: "",
    area: "",
    town: ""
  },
  village: "",
  carrierCode: "",
  disaster: {
    major: "",
    minor: "",
    index: ""
  },
  desc: ""
});

const minorOptions = {
  "1": [{ label: "震情信息", value: "01" }],
  "2": [
    { label: "死亡", value: "01" },
    { label: "受伤", value: "02" },
    { label: "失踪", value: "03" }
  ],
  "3": [
    { label: "土木", value: "01" },
    { label: "砖木", value: "02" },
    { label: "砖混", value: "03" },
    { label: "框架", value: "04" },
    { label: "其他", value: "05" }
  ],
  "4": [
    { label: "交通", value: "01" },
    { label: "供水", value: "02" },
    { label: "输油", value: "03" },
    { label: "燃气", value: "04" },
    { label: "电力", value: "05" },
    { label: "通信", value: "06" },
    { label: "水利", value: "07" }
  ],
  "5": [
    { label: "崩塌", value: "01" },
    { label: "滑坡", value: "02" },
    { label: "泥石流", value: "03" },
    { label: "岩溶塌陷", value: "04" },
    { label: "地裂缝", value: "05" },
    { label: "地面沉降", value: "06" },
    { label: "其他", value: "07" }
  ]
};
const indexOptions = {
  "1": [
    { label: "地理位置", value: "001" },
    { label: "时间", value: "002" },
    { label: "震级", value: "003" },
    { label: "震源深度", value: "004" },
    { label: "列度", value: "005" }
  ],
  "2": [
    { label: "受灾人数", value: "001" },
    { label: "受灾程度", value: "002" }
  ],
  "3": [
    { label: "一般损坏面积", value: "001" },
    { label: "严重损坏面积", value: "002" },
    { label: "受灾程度", value: "003" }
  ],
  "4": [
    { label: "受灾设施数", value: "001" },
    { label: "受灾范围", value: "002" },
    { label: "受灾程度", value: "003" }
  ],
  "5": [
    { label: "灾害损失", value: "001" },
    { label: "灾害范围", value: "002" },
    { label: "受灾程度", value: "003" }
  ]
};
const onChangeDisasterMajor = value => {
  form.disaster.minor = minorOptions[value][0].value;
  form.disaster.index = indexOptions[value][0].value;
};

const villageOptions = reactive([]);
const onChangeRegion = async () => {
  if (form.location.town) {
    const param = {
      api_key: "823584b00b661041a21a4f4139a4c2bbc60d",
      fields: "village_code,division_name",
      levels: "5",
      townCodes: form.location.town + "000",
      page: "1",
      size: "50"
    };
    const response = (await http.request(
      "get",
      "https://api.apihubs.cn/administrativeDivision/get",
      { params: param }
    )) as any;
    villageOptions.splice(0);
    console.log(response);
    response.data.list.forEach(element => {
      villageOptions.push({
        title: element.division_name,
        value: element.village_code.toString()
      });
    });
    form.village = villageOptions[0].value;
    console.log(villageOptions);
  } else {
    form.village = "";
    villageOptions.splice(0);
  }
};

const upload = ref<UploadInstance>();
const handleExceed: UploadProps["onExceed"] = files => {
  upload.value!.clearFiles();
  const file = files[0] as UploadRawFile;
  file.uid = genFileId();
  upload.value!.handleStart(file);
};
const submitUpload = () => {
  upload.value!.submit();
};
const onChangeFile = (file, fileList) => {
  form.desc = file.name;
  console.log(fileList[0]);
};

const onSubmit = async () => {
  const response = (await http.request("post", "/api/event/add", {
    params: form
  })) as any;
  if (response.status == 200) {
    ElMessage({
      message: "提交成功",
      type: "success"
    });
  } else {
    ElMessage.error("提交失败，请通过console查看原因");
    console.log(response);
  }
};
</script>
