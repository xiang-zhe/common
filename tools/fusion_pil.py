from PIL import Image
 
 
def blend_two_images():
    img1 = Image.open( "bridge.png ")
    img1 = img1.convert('RGBA')
 
    img2 = Image.open( "birds.png ")
    img2 = img2.convert('RGBA')
    
    img = Image.blend(img1, img2, 0.3)
    img.show()
    img.save( "blend.png")
 
    return
    
    
def blend_two_images2():
    img1 = Image.open( "bridge.png ")
    img1 = img1.convert('RGBA')
 
    img2 = Image.open( "birds.png ")
    img2 = img2.convert('RGBA')
    
    r, g, b, alpha = img2.split()
    alpha = alpha.point(lambda i: i>0 and 204) #类似Image.blend（，，0.3）
 
    img = Image.composite(img2, img1, alpha)
 
    img.show()
    img.save( "blend2.png")
 
    return
