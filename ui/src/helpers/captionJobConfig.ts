import { CaptionJobConfig } from "@/types";
import { captionerTypes, defaultImageCaptionPrompt } from "./captionOptions";


export const defaultCaptionJobConfig: CaptionJobConfig = {
  job: 'extension',
  config: {
    name: 'Caption Directory',
    process: [
      {
        type: 'Qwen3VLCaptioner',
        sqlite_db_path: './aitk_db.db',
        device: 'cuda',
        caption: {
          model_name_or_path: "Qwen/Qwen3-VL-8B-Instruct",
          dtype: 'bf16',
          quantize: true,
          qtype: 'float8',
          low_vram: true,
          extensions: ['jpg', 'jpeg', 'png', 'bmp', 'webp', 'jxl'],
          path_to_caption: '',
          recaption: false,
          caption_prompt: defaultImageCaptionPrompt,
          max_res: 512,
          max_new_tokens: 128,
        },
      },
    ],
  },
};


const repairDefaults = (defaults: { [key: string]: any }) => {
  let newDefaults: { [key: string]: any } = {};
  // if the key doesnt start with config.process[0]., then add it
  for (const key in defaults) {
    if (!key.startsWith('config.process[0].')) {
      newDefaults[`config.process[0].${key}`] = defaults[key];
    } else {
      newDefaults[key] = defaults[key];
    }
  }
  return newDefaults;
}



export const handleCaptionerTypeChange = (
  currentTypeName: string,
  newTypeName: string,
  jobConfig: CaptionJobConfig,
  setJobConfig: (value: any, key: string) => void,
) => {
  const currentType = captionerTypes.find(a => a.name === currentTypeName);
  if (!currentType || currentType.name === newTypeName) {
    return;
  }

  // update the defaults when a model is selected
  const newType = captionerTypes.find(model => model.name === newTypeName);

  let currentDefaults = repairDefaults(currentType.defaults || {});
  let newDefaults = repairDefaults(newType?.defaults || {});

  // set new model
  setJobConfig(newTypeName, 'config.process[0].type');

  // revert defaults from previous model
  for (const key in currentDefaults) {
    setJobConfig(currentDefaults[key][1], key);
  }

  for (const key in newDefaults) {
    setJobConfig(newDefaults[key][0], key);
  }
};
