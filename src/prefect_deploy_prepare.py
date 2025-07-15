from prefect_prepare_data import prefect_prepare_data_flow

if __name__ == "__main__":
    prefect_prepare_data_flow.serve(
        name="prepare-data-every-5min",
        cron="*/5 * * * *",
    )
