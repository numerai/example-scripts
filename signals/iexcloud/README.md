Let's follow the current De-Fi hype... APY APY APY. The idea is very simple: what is the annual percentage yield for a stock, based on its price and dividends?

**Before running, install the python packages for iexfinance and numerapi:** `pip install iexfinance numerapi`

Additionally, IEX_TOKEN, NUMERAI_PUBLIC_ID, and NUMERAI_SECRET_KEY need to be set on lines 13-16.

To get an "IEX_TOKEN", create an account on [IEX Cloud](https://iexcloud.io/). Make sure you switch the toggle to "sandbox" before copying the token.

## Details
There's a nice endpoint on IEX Cloud's API: [https://iexcloud.io/docs/api/#dividends-basic](https://iexcloud.io/docs/api/#dividends-basic). The fields `amount` and `frequency`, combined with the daily price, being all we need.

IEX Cloud is free-ish, meaning there is a free account but it's limited in endpoints and calls per month. However, they also provide a sandbox mode which you can call as much as you want. Some data in sandbox mode is obfuscated but not all. Also, a subscription for $9/month (~0.15NMR?!) is not the end of the world.

Anyway there's two Python wrappers for the API: [https://github.com/addisonlynch/iexfinance](https://github.com/addisonlynch/iexfinance) and [https://github.com/iexcloud/pyEX](https://github.com/iexcloud/pyEX).

Results for this rough version are not bad: [https://signals.numer.ai/gosuto_test](https://signals.numer.ai/gosuto_test)

![Screenshot 2021-04-23 at 23.24.35|214x500](https://forum.numer.ai/uploads/default/optimized/1X/795a9d06cf177448114f8eb10ce8ca1768570dd6_2_321x750.png)
