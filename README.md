[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Solver
====

## Overview

Linear Solver

## License

These codes are licensed under Apache License 2.0.

## ISSUE & TODO
- [x]vpcrのアルゴリズムがおかしい．
- [o]gcr->done, kskipcg->done, kskipbicg->done, vpgcr->done, には二次元配列で格納したデータがあります，それを全部一次元配列に統一することを考えている．
- [o]gmresの並列化
- [o]SolverName not found

## MEMO
MultiGPU 行列ベクトル積　実装

普通　CG CR GCR VPCG VPGCR
転置　BICG 
Base　KSKIPCG KSKIPBICG

!! GCR and VPGCR's Scalar_axy (in Gcr_sp )calling is slow
!! As same as GCR, GMRES and VPGMRES 's Gmres_sp is slow


## All Solver
cg cr gcr bicg kskipcg kskipbicg gmres vpcg vpcr[BUG] vpgcr vpgmres

## 未実装
  KSKIPCR, VPBICG, VPCR

