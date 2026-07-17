# Crystal Generation Sandbox

Проект представляет собой унифицированную обёртку для пяти генеративных моделей кристаллических материалов: ADiT, WyFormer, MiAD, SGEquiDiff, CrystalDiT. Все модели запускаются одной командой python run.py <model>. 
Установка: клонируйте репозиторий, создайте conda-окружение crystal_sandbox с Python 3.10, установите зависимости из requirements.txt. Поместите оригинальные репозитории в папку external/ (симлинки). Для каждой модели установите её зависимости в отдельном conda-окружении согласно её README.

Запуск: python run.py <model>, где model: adit, wyformer, miad, sgequidiff, crystaldit. Результаты сохраняются в стандартные для каждой модели папки.
