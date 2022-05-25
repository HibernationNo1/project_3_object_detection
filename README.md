### directory map

**`labelme(dir)`**

- `annotations(dir)` 

  - `{category_dataset}_1(dir)`

  - `{category_dataset}_2(dir)`

    ...

  - `{category_dataset}_N(dir)`

    - `image_1.jpg`

    - `image_1.json`

      ...

    - `image_N.jpg`

    - `image_N.json`

- `train_dataset(dir)`

  - `{yyyy-mm-dd}_{category_dataset}_1(dir)`

  - `{yyyy-mm-dd}_{category_dataset}_2(dir)`

    ...

  - `{yyyy-mm-dd}_{category_dataset}_N(dir)`

    - `dataset(dir)`

      - `image_1.jpg`

      - `image_2.jpg`

        ...

      - `image_3.jpg`

      - `{yyyy_mm_dd}_{category_dataset}_train_dataset.json`

    - `gt_images(dir)`   *option*

**`configs`**

- `lebelme_config.py`
- 

**`labelme.py`**



