import cdsapi

c = cdsapi.Client()

# u
for yr in range(2024, 2026):
    for mo in range(1, 13):

        print(f'Downloading ERA5 U for {yr}_{mo:02d}')
        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                "product_type": ["reanalysis"],
                "variable": ["u_component_of_wind"],
                "data_format": "netcdf",
                "download_format": "unarchived",
                'year': [str(yr)],
                'month': [
                    f'{mo:02d}'
                ],
                'day': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                    '13', '14', '15',
                    '16', '17', '18',
                    '19', '20', '21',
                    '22', '23', '24',
                    '25', '26', '27',
                    '28', '29', '30',
                    '31',
                ],
                'time': [
                    '00:00', '06:00', '12:00', '18:00',
                ],
                "pressure_level": [
                    "1", "2", "3",
                    "5", "7", "10",
                    "20", "30", "50",
                    "70", "100", "125",
                    "150", "175", "200",
                    "225", "250", "300",
                    "350", "400", "450",
                    "500", "550", "600",
                    "650", "700", "750",
                    "775", "800", "825",
                    "850", "875", "900",
                    "925", "950", "975",
                    "1000"
                ]
            },
            f'/scratch/bell/hu1029/Data/raw/u_component_of_wind_{yr}_{mo:02d}.nc')
        print(f'Done downloading ERA5 U for {yr} -------------------')

# v
for yr in range(2024, 2026):
    for mo in range(1, 13):

        print(f'Downloading ERA5 V for {yr}_{mo:02d}')
        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                "product_type": ["reanalysis"],
                "variable": ["v_component_of_wind"],
                "data_format": "netcdf",
                "download_format": "unarchived",
                'year': [str(yr)],
                'month': [
                    f'{mo:02d}'
                ],
                'day': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                    '13', '14', '15',
                    '16', '17', '18',
                    '19', '20', '21',
                    '22', '23', '24',
                    '25', '26', '27',
                    '28', '29', '30',
                    '31',
                ],
                'time': [
                    '00:00', '06:00', '12:00', '18:00',
                ],
                "pressure_level": [
                    "1", "2", "3",
                    "5", "7", "10",
                    "20", "30", "50",
                    "70", "100", "125",
                    "150", "175", "200",
                    "225", "250", "300",
                    "350", "400", "450",
                    "500", "550", "600",
                    "650", "700", "750",
                    "775", "800", "825",
                    "850", "875", "900",
                    "925", "950", "975",
                    "1000"
                ]
            },
            f'/scratch/bell/hu1029/Data/raw/v_component_of_wind_{yr}_{mo:02d}.nc')
        print(f'Done downloading ERA5 V for {yr} -------------------')
