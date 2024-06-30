import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Sample data for fitting the encoder (should be the same data used during training)
Brand = np.array(['Samsung', 'LG', 'MI', 'Sony Bravia', 'Panasonic', 'TCL', 'Sony Bravia2', 'Hisense', 'Acer', 'TOSHIBA', 'OnePlus','Haier']).reshape(-1, 1)
Screen = np.array(['LED', 'OLED', 'QLED', 'QNED', 'Laser','NanoCell']).reshape(-1, 1)
Display = np.array(['4K UHD', '4K', '8K UHD', '4K Dolby Vision','Full HD','HD','UHD']).reshape(-1, 1)
Platform = np.array(['Smart TV', 'Android', 'Google TV', 'Fire TV']).reshape(-1, 1)

# Fit the encoders
ohe_Brand = OneHotEncoder(handle_unknown='ignore')
ohe_Screen = OneHotEncoder(handle_unknown='ignore')
ohe_Display = OneHotEncoder(handle_unknown='ignore')
ohe_Platform = OneHotEncoder(handle_unknown='ignore')

ohe_Brand.fit(Brand)
ohe_Screen.fit(Screen)
ohe_Display.fit(Display)
ohe_Platform.fit(Platform)

# Save the categories to .npy files
np.save('Transformation/onehot_Brand_encoder.npy', ohe_Brand.categories_)
np.save('Transformation/onehot_Screen_encoder.npy', ohe_Screen.categories_)
np.save('Transformation/onehot_Display_encoder.npy', ohe_Display.categories_)
np.save('Transformation/onehot_Platform_encoder.npy', ohe_Platform.categories_)
