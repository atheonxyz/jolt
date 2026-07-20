/// Mirrors the Noir `date` library's behavior for age/expiry checks.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimpleDate {
    pub year: u32,
    pub month: u32,
    pub day: u32,
}

impl SimpleDate {
    /// Convert a Unix timestamp (seconds since 1970-01-01) to a date.
    /// Uses Howard Hinnant's civil_from_days algorithm.
    pub fn from_timestamp(ts: u64) -> Self {
        let days = (ts / 86400) as i64;
        // Shift epoch from 0000-03-01 to 1970-01-01
        let z = days + 719468;
        let era = if z >= 0 { z } else { z - 146096 } / 146097;
        let doe = (z - era * 146097) as u32; // day of era [0, 146096]
        let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // year of era [0, 399]
        let y = yoe as i64 + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // day of year [0, 365]
        let mp = (5 * doy + 2) / 153; // month index [0, 11]
        let d = doy - (153 * mp + 2) / 5 + 1; // day [1, 31]
        let m = if mp < 10 { mp + 3 } else { mp - 9 }; // month [1, 12]
        let y = if m <= 2 { y + 1 } else { y };

        SimpleDate {
            year: y as u32,
            month: m,
            day: d,
        }
    }

    /// Parse a 6-byte YYMMDD (ASCII digits) into a date.
    /// Uses `pivot_year` to disambiguate century:
    /// if 2000+YY > pivot_year, interpret as 19xx; else 20xx.
    /// This matches the Noir `from_bytes_short_year` behavior.
    pub fn from_yymmdd_bytes(bytes: &[u8], pivot_year: u32) -> Self {
        let yy = ascii_digit(bytes[0]) * 10 + ascii_digit(bytes[1]);
        let mm = ascii_digit(bytes[2]) * 10 + ascii_digit(bytes[3]);
        let dd = ascii_digit(bytes[4]) * 10 + ascii_digit(bytes[5]);

        let full_year = if 2000 + yy > pivot_year {
            1900 + yy
        } else {
            2000 + yy
        };

        SimpleDate {
            year: full_year,
            month: mm,
            day: dd,
        }
    }

    /// Return a new date with `n` years added. Same month/day.
    pub fn add_years(self, n: u32) -> Self {
        SimpleDate {
            year: self.year + n,
            month: self.month,
            day: self.day,
        }
    }

    /// self >= other (greater than or equal)
    pub fn gte(self, other: Self) -> bool {
        (self.year, self.month, self.day) >= (other.year, other.month, other.day)
    }

    /// self < other (strictly less than)
    pub fn lt(self, other: Self) -> bool {
        (self.year, self.month, self.day) < (other.year, other.month, other.day)
    }
}

fn ascii_digit(b: u8) -> u32 {
    (b - b'0') as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_timestamp_epoch() {
        let d = SimpleDate::from_timestamp(0);
        assert_eq!(
            d,
            SimpleDate {
                year: 1970,
                month: 1,
                day: 1
            }
        );
    }

    #[test]
    fn test_from_timestamp_prover_toml() {
        // 1758300539 is the timestamp from the Prover.toml
        let d = SimpleDate::from_timestamp(1758300539);
        // 2025-09-19 (approximately)
        assert_eq!(d.year, 2025);
        assert_eq!(d.month, 9);
        assert_eq!(d.day, 19);
    }

    #[test]
    fn test_from_yymmdd_bytes() {
        // "070101" = Jan 1, 2007 (with pivot 2025)
        let bytes = [b'0', b'7', b'0', b'1', b'0', b'1'];
        let d = SimpleDate::from_yymmdd_bytes(&bytes, 2025);
        assert_eq!(
            d,
            SimpleDate {
                year: 2007,
                month: 1,
                day: 1
            }
        );
    }

    #[test]
    fn test_from_yymmdd_bytes_pivot_19xx() {
        // "950101" = Jan 1, 1995 (with pivot 2025, since 2095 > 2025)
        let bytes = [b'9', b'5', b'0', b'1', b'0', b'1'];
        let d = SimpleDate::from_yymmdd_bytes(&bytes, 2025);
        assert_eq!(
            d,
            SimpleDate {
                year: 1995,
                month: 1,
                day: 1
            }
        );
    }

    #[test]
    fn test_add_years() {
        let d = SimpleDate {
            year: 2000,
            month: 6,
            day: 15,
        };
        let d2 = d.add_years(18);
        assert_eq!(
            d2,
            SimpleDate {
                year: 2018,
                month: 6,
                day: 15
            }
        );
    }

    #[test]
    fn test_gte_lt() {
        let a = SimpleDate {
            year: 2025,
            month: 9,
            day: 19,
        };
        let b = SimpleDate {
            year: 2018,
            month: 6,
            day: 15,
        };
        assert!(a.gte(b));
        assert!(!a.lt(b));
        assert!(b.lt(a));
        assert!(!b.gte(a));
        assert!(a.gte(a)); // equal case
    }
}
