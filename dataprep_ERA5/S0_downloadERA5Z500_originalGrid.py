import cdsapi

c = cdsapi.Client()

for yr in range(2022, 2026):

    print(f'Downloading ERA5 Z500 for {yr}')
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            "product_type": ["reanalysis"],
            "variable": ["geopotential"],
            "pressure_level": ["500"],
            "data_format": "netcdf",
            "download_format": "unarchived",
            'year': [str(yr)],
            'month': [
                '01', '02', '03', '04', '05', '06',
                '07', '08', '09', '10', '11', '12',
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
            ]
        },
        f'/scratch/bell/hu1029/Data/raw/ERA5_Z500_originalGrid/ERA5_Z500_6hr_{yr}.nc')
    print(f'Done downloading ERA5 Z500 for {yr} -------------------')
