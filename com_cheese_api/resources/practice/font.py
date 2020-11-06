import matplotlib.font_manager
font_list = matplotlib.font_manager.findSystemFonts(fontpaths = None, fontext='ttf')

# ttf 폰트 전체개수
#print(len(font_list))

# 시스템 폰트에서 읽어온 리스트에서 상위 10개만 출력
print(font_list[:10])
