
def GID15():
    class_names = ['industrial_land', 'urban_residential', 'rural_residential', 'traffic_land', 'paddy_field',
                   'irrigated_land', 'dry_cropland', 'garden_plot', 'arbor_woodland', 'shrub_land',
                   'natural_grassland', 'artificial_grassland', 'river', 'lake', 'pond']

    palette = [[200, 0, 0], [250, 0, 150], [200, 150, 150], [250, 150, 150], [0, 200, 0],
               [150, 250, 0], [150, 200, 150], [200, 0, 200], [150, 0, 250], [150, 150, 250],
               [250, 200, 0], [200, 200, 0], [0, 0, 200], [0, 150, 200], [0, 200, 250]]

    return class_names, palette


def iSAID():
    class_names = ['Ship', 'Storage_Tank', 'Baseball_Diamond', 'Tennis_Court', 'Basketball_Court',
                   'Ground_Track_Field', 'Bridge', 'Large_Vehicle', 'Small_Vehicle', 'Helicopter',
                   'Swimming_Pool', 'Roundabout','Soccer_Ball_Field', 'Plane', 'Harbor']

    palette = [[0, 0, 63], [0, 63, 63], [0, 63, 0], [0, 63, 127], [0, 63, 191],
               [0, 63, 255], [0, 127, 63], [0, 127, 127], [0, 0, 127], [0, 0, 191],
               [0, 0, 255], [0, 191, 127], [0, 127, 191], [0, 127, 255], [0, 100, 155]]

    return class_names, palette


def MARS():
    class_names = ['Martian Soil', 'Sands', 'Gravel', 'Bedrock', 'Rocks',
                   'Tracks', 'Shadows', 'Unknown', 'Background']

    palette = [[128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128],
               [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0]]

    return class_names, palette


def Vaihingen():
    class_names = ['Impervious_Surface', 'Building', 'Low_Vegetation', 'Tree', 'Car']

    palette = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],[255, 255, 0]]

    return class_names, palette


def DFC22():
    class_names = ['Urban fabric', 'Industrial', 'Mine', 'Artificial', 'Arable', 'Permanent crops',
                   'Pastures', 'Forests', 'Herbaceous', 'Open spaces', 'Wetlands', 'Water']

    palette = [[219, 95, 87], [219, 151, 87], [219, 208, 87], [173, 219, 87], [117, 219, 87], [123, 196, 123],
               [88, 177, 88], [0, 128, 0], [88, 176, 167], [153, 93, 19], [87, 155, 219], [0, 98, 255]]

    return class_names, palette