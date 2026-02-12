from pathlib import Path
import matplotlib.pyplot as plt

style_file = "styles/professional_style.mplstyle"
style_path = Path(__file__).parent / style_file
plt.style.use(style_path)