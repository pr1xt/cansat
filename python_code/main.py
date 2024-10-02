from afforestation import AfforestationArea


def main():
    afforestation = AfforestationArea('../assets/map2km.png')
    area = afforestation.calculate_area()
    afforestation.display_images()

    print(f"Total area in acres: {area:.2f}")


if __name__ == '__main__':
    main()
