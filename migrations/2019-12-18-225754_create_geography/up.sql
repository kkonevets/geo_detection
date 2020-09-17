-- Your SQL goes here

CREATE TABLE public.geography
(
    extid bigint NOT NULL,
    profile_proc_location text COLLATE pg_catalog."default",
    profile_proc_city text COLLATE pg_catalog."default",
    profile_proc_hometown text COLLATE pg_catalog."default",
    CONSTRAINT geography_pkey PRIMARY KEY (extid)
)

TABLESPACE pg_default;
