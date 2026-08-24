# test-repo

**Propósito:** proyecto de prueba (smoke test) del framework de training
refactorizado (TrainingSchema + DataProvider + TrainingOrchestrator). NO
es un proyecto de investigación real -- sirve para validar que el
orquestador nuevo funciona correctamente contra las estrategias
implementadas, antes de usarlas en proyectos reales como NID.

## Qué se prueba acá

| `strategy_name` | Schema | Qué valida |
|---|---|---|
| `WS_Holdout` | `WithinUnitHoldoutSchema` | Split online-simulado 60/20/20 |
| `WS_KFold` | `WithinUnitKFoldSchema` | K-fold aleatorio dentro de sujeto |
| `WS_LOSO` | `LeaveOneUnitOutSchema` | Leave-One-Subject-Out |

## Dataset de prueba

Cho2017, sujetos `[3, 13, 14, 26, 34, 35, 37, 43]` -- subconjunto chico
elegido para que las corridas de prueba sean rápidas, no representativo
de resultados de investigación.

## Cómo correr

```bash
python scripts/run_production.py \
  --config projects/test-repo/configs/training/WS_Holdout/test-repo_WS_Holdout_EEGNetv4_Cho2017_smoke-test.json
```

## Nota

Una vez validado el framework acá, los proyectos reales (ej. `NID/`)
usan la misma estructura de config -- este proyecto no se publica ni se
mantiene como parte de ningún paper.