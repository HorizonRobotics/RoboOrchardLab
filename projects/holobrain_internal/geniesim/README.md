# GenieSim Evaluation

## 1. GenieSim Third-Party Source Integration

This directory manages a lightweight integration of the upstream **genie_sim** repository.

### Usage

Run the following steps to fetch the upstream code and apply local patches:

```bash
cd projects/holobrain/geniesim/3rdparty
bash clone_geniesim.sh
```

This script will:

- Fetch the required upstream commit and keep only the `source/` directory
- Apply all local patches stored in the `patches/` folder
